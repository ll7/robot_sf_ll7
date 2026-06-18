"""Policy metadata helpers for map-based benchmark runs."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def holonomic_world_velocity_command(vx: float, vy: float) -> dict[str, float | str]:
    """Build an explicit world-frame holonomic velocity command payload.

    Returns:
        dict[str, float | str]: Structured command payload for holonomic env actions.
    """
    return {
        "command_kind": "holonomic_vxy_world",
        "vx": float(vx),
        "vy": float(vy),
    }


def apply_direct_world_velocity_metadata(
    meta: dict[str, Any],
    *,
    adapter_boundary: str | None = None,
) -> None:
    """Mark planner metadata as direct world-velocity execution for holonomic benchmarks."""
    planner_meta = meta.get("planner_kinematics")
    if isinstance(planner_meta, dict):
        planner_meta["planner_command_space"] = "holonomic_vxy_world"
        planner_meta["benchmark_command_space"] = "holonomic_vxy_world"
        planner_meta["projection_policy"] = "world_velocity_passthrough"
        planner_meta["execution_detail"] = "direct_holonomic_world_velocity"
    upstream_reference = meta.get("upstream_reference")
    if isinstance(upstream_reference, dict) and adapter_boundary is not None:
        upstream_reference["adapter_boundary"] = adapter_boundary


def attach_planner_reset(policy: Callable[..., Any], adapter: Any) -> None:
    """Attach a planner reset hook that tolerates adapters without seed support."""
    reset = getattr(adapter, "reset", None)
    if not callable(reset):
        return
    try:
        reset_accepts_seed = _callable_accepts_seed(reset)
    except (TypeError, ValueError):
        reset_accepts_seed = True

    def _planner_reset(seed: int | None = None) -> None:
        """Reset an adapter, using seed-aware reset when supported."""
        if seed is None or not reset_accepts_seed:
            reset()
            return
        reset(seed=seed)

    policy._planner_reset = _planner_reset


def _callable_accepts_seed(func: Callable[..., Any]) -> bool:
    """Return whether a callable can bind a ``seed=...`` keyword argument."""
    signature = inspect.signature(func)
    try:
        signature.bind(seed=0)
    except TypeError:
        return False
    return True


def finalize_feasibility_metadata(meta: dict[str, Any]) -> None:
    """Finalize per-episode feasibility rates/means and strip internal accumulators."""
    feasibility = meta.get("kinematics_feasibility")
    if not isinstance(feasibility, dict):
        return
    total = int(feasibility.get("commands_evaluated", 0))
    infeasible = int(feasibility.get("infeasible_native_count", 0))
    projected = int(feasibility.get("projected_count", 0))
    sum_linear = float(feasibility.pop("_sum_abs_delta_linear", 0.0))
    sum_angular = float(feasibility.pop("_sum_abs_delta_angular", 0.0))
    max_linear = float(feasibility.pop("_max_abs_delta_linear", 0.0))
    max_angular = float(feasibility.pop("_max_abs_delta_angular", 0.0))
    if total > 0:
        feasibility["projection_rate"] = float(projected / total)
        feasibility["infeasible_rate"] = float(infeasible / total)
        feasibility["mean_abs_delta_linear"] = float(sum_linear / total)
        feasibility["mean_abs_delta_angular"] = float(sum_angular / total)
    else:
        feasibility["projection_rate"] = 0.0
        feasibility["infeasible_rate"] = 0.0
        feasibility["mean_abs_delta_linear"] = 0.0
        feasibility["mean_abs_delta_angular"] = 0.0
    feasibility["max_abs_delta_linear"] = max_linear
    feasibility["max_abs_delta_angular"] = max_angular
