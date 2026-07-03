"""Builder for TAGA-like social-group avoidance map-runner policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.map_runner_policy_common import build_adapter_policy
from robot_sf.planner.group_avoidance import (
    CLAIM_BOUNDARY,
    TangentSubgoalGroupAvoidanceAdapter,
    build_group_avoidance_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable


GROUP_AVOIDANCE_ALGO_KEYS = frozenset({"taga_group_avoidance", "group_avoidance"})


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build the diagnostic TAGA-like tangent-subgoal wrapper.

    Returns:
        tuple: Policy callable and enriched metadata.
    """

    del adapter_impact_eval
    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    config = build_group_avoidance_config(algo_config)
    adapter = TangentSubgoalGroupAvoidanceAdapter(config=config)
    meta: dict[str, Any] = {
        "group_avoidance": {
            "schema_version": "taga-like-group-avoidance.v1",
            "status": "enabled",
            "diagnostic_only": True,
            "wrapped_algo": config.wrapped_algo,
            "trigger_mode": config.trigger_mode,
            "safety_margin_m": config.safety_margin_m,
            "tangent_clearance_m": config.tangent_clearance_m,
            "tangent_side": config.tangent_side,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    }
    return build_adapter_policy(
        algo_key=algo_key,
        algo_config=algo_config,
        meta=meta,
        adapter=adapter,
        adapter_name="TangentSubgoalGroupAvoidanceAdapter",
        robot_kinematics=robot_kinematics,
        normalized_robot_command_mode=normalized_robot_command_mode,
        limitations="diagnostic group-space wrapper; group-intrusion only, not safety evidence",
    )


__all__ = ["GROUP_AVOIDANCE_ALGO_KEYS", "build"]
