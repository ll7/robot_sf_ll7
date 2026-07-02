"""Builder for route-conditioned learned local planner policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.map_runner_policy_common import build_adapter_policy
from robot_sf.planner.hybrid_global_rl import (
    HybridGlobalRLLocalAdapter,
    build_hybrid_global_rl_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable

HYBRID_GLOBAL_RL_KEYS = frozenset(
    {"hybrid_global_rl", "global_rl_local", "route_conditioned_rl", "hybrid_route_rl"}
)


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build route-conditioned learned local planner policy metadata.

    Returns:
        Policy callable and enriched map-runner metadata.
    """

    if algo_key not in HYBRID_GLOBAL_RL_KEYS:
        supported = ", ".join(sorted(HYBRID_GLOBAL_RL_KEYS))
        raise ValueError(
            f"Unsupported hybrid_global_rl algo_key {algo_key!r}; expected {supported}"
        )
    del adapter_impact_eval
    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    config = build_hybrid_global_rl_config(algo_config)
    adapter = HybridGlobalRLLocalAdapter(config=config)
    meta: dict[str, Any] = {
        "algorithm": algo_key,
        "hybrid_global_rl": {
            "status": "enabled",
            "waypoint_provider": config.waypoint_provider,
            "local_policy_algo": config.local_policy_algo,
            "allow_goal_fallback": config.allow_goal_fallback,
            "claim_boundary": "diagnostic-only route-conditioned RL local planner",
        },
    }
    return build_adapter_policy(
        algo_key=algo_key,
        algo_config=algo_config,
        meta=meta,
        adapter=adapter,
        adapter_name="HybridGlobalRLLocalAdapter",
        robot_kinematics=robot_kinematics,
        normalized_robot_command_mode=normalized_robot_command_mode,
        limitations="diagnostic_only_not_benchmark_evidence",
    )
